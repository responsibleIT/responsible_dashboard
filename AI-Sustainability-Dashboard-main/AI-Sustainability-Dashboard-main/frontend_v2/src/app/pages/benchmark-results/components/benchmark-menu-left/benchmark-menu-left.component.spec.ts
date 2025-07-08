import { ComponentFixture, TestBed } from '@angular/core/testing';

import { BenchmarkMenuLeftComponent } from './benchmark-menu-left.component';

describe('BenchmarkMenuLeftComponent', () => {
  let component: BenchmarkMenuLeftComponent;
  let fixture: ComponentFixture<BenchmarkMenuLeftComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [BenchmarkMenuLeftComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(BenchmarkMenuLeftComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
