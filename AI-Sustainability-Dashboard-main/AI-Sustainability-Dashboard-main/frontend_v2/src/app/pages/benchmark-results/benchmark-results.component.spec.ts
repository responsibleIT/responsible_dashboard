import { ComponentFixture, TestBed } from '@angular/core/testing';

import { BenchmarkResultsComponent } from './benchmark-results.component';

describe('ValidationResultsComponent', () => {
  let component: BenchmarkResultsComponent;
  let fixture: ComponentFixture<BenchmarkResultsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [BenchmarkResultsComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(BenchmarkResultsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
