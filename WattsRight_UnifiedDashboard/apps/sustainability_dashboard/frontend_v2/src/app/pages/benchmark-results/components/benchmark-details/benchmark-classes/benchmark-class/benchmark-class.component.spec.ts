import { ComponentFixture, TestBed } from '@angular/core/testing';

import { BenchmarkClassComponent } from './benchmark-class.component';

describe('BenchmarkClassComponent', () => {
  let component: BenchmarkClassComponent;
  let fixture: ComponentFixture<BenchmarkClassComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [BenchmarkClassComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(BenchmarkClassComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
